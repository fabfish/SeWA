import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import copy


def load_and_average_models(model, model_list, steps, last_model=None, add_decay=True):
    target = copy.deepcopy(model)

    average_params = None
    last_model_params = None
    count = 0

    for model_params, i in zip(model_list, steps):
        
        # model_params = model.state_dict()
        if average_params is None:
            # 初始化平均参数
            average_params = {
                key: paddle.zeros_like(value, dtype='float32')
                for key, value in model_params.items() if paddle.is_floating_point(value)
            }

        # 累加参数
        for key in model_params:
            if key in average_params:
                average_params[key] += model_params[key].astype('float32')

        last_model_params = model_params
        count += 1
        print(f"Successfully loaded and added model {i}")
    
    if last_model is not None:
        last_model_params = last_model

    # 计算平均值
    if count > 0:
        for key in average_params:
            average_params[key] /= count
        # 对非 float 类型的参数，使用最后一个模型的值
        for key, value in last_model_params.items():
            if key not in average_params:
                average_params[key] = value
        print("Successfully averaged model parameters.")
    else:
        print("No models were loaded for averaging.")

    if add_decay:
        decay = 0.9
        for name, param in last_model_params.items():
            if name in last_model_params and paddle.is_floating_point(param):
                average_params[name] = average_params[name] * decay + param * (1 - decay)
            else:
                average_params[name] = param.clone()

    target.set_state_dict(average_params)
    return target

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)
    return mod

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, model=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            # assert pretrained_checkpoint is not None
            with paddle.no_grad():
                # print('TaskVector:' + pretrained_checkpoint)
                model.set_state_dict(pretrained_checkpoint)
                # pretrained_state_dict = model.state_dict()
                self.vector = {}
                for key, value in model.named_parameters():
                    # if value.dtype in [paddle.int64, paddle.uint8]:
                    #     continue
                    self.vector[key] = value
    
    def __add__(self, other):
        """Add two task vectors together."""
        with paddle.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with paddle.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def weightmerging(self, taskvectors, coefficients):
        with paddle.no_grad():
            new_vector = {}
            for key in taskvectors[0].vector:
                new_vector[key] = sum(coefficients[k] * taskvectors[k][key] for k in range(len(taskvectors)))
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with paddle.no_grad():
            pretrained_model = paddle.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.set_state_dict(new_state_dict, use_structured_name=False)
        return pretrained_model

class MergeNet(nn.Layer):
    def __init__(self, model=None, model_list=None, temperature=1.0, k=10):
        super(MergeNet, self).__init__()

        self.model = model
        self.model_list = model_list
        self.infer_model = copy.deepcopy(model).eval()
        for p in self.infer_model.parameters():
            p.stop_gradient = True
        self.n = len(model_list)
        self.temperature = temperature
        self.k = k

        task_vectors = []
        for model_state_dict in self.model_list:
            task_vectors.append(TaskVector(model_state_dict, model))

        _, names = make_functional(model)
        self.names = names
        non_params = {}
        for k, v in self.infer_model.state_dict().items():
            if 'bn' in k:
                non_params[k] = v
        self.non_params = non_params

        self.mask_logit = self.create_parameter(shape=[self.n], default_initializer=nn.initializer.Constant(0.0))
        # random_idx = paddle.randperm(self.n)[:int(k / 2)]
        step = int(self.n / self.k)
        self.mask_logit.set_value(paddle.zeros([self.n]))
        self.mask_logit[::step] = 0.001

        paramslist = []
        shape_list = []
        itx = [0]
        for k, v in task_vectors[0].vector.items():
            shape_list.append(v.shape)
            itx.append(itx[-1] + v.flatten().shape[0])
        paramslist += [[v.detach().flatten() for _, v in tv.vector.items()] for i, tv in enumerate(task_vectors)] # task vectors
        self.paramslist = paramslist
        self.params_tensor = paddle.stack([paddle.concat(p) for p in self.paramslist])
        self.shape_list = shape_list
        self.itx = itx

    def collect_trainable_params(self):
        return [self.mask_logit]
    
    def gumbel_softmax(self, temperature=1.0, eps=1e-9):
        u = torch.rand_like(self.mask_logit)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        y = self.mask_logit + gumbel
        y = F.sigmoid(y / temperature)

        y_hard = torch.zeros_like(y)
        y_hard[y > 0.5] = 1.0

        return (y_hard - y).detach() + y

    def project_logit(self):
        with torch.no_grad():
            k_th_lagest = torch.topk(self.mask_logit, self.k)[0][-1]
            self.mask_logit.data[self.mask_logit < k_th_lagest] = -1e9

    def compute_mask(self, eval=False):
        if not eval:
            mask = self.gumbel_softmax(self.temperature)
        else:
            topk_values, topk_indices = torch.topk(self.mask_logit, self.k)
            mask = torch.zeros_like(self.mask_logit)
            mask[topk_indices] = 1
        return mask

    def forward(self, x, eval=False):
        mask = self.compute_mask(eval=eval)
        # params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, mask))) / self.k for j, p in enumerate(zip(*self.paramslist)))
        # params = tuple(p.cuda() for p in params)
        params = torch.sum(mask.unsqueeze(-1) * self.params_tensor, dim=0) / self.k
        recover = []
        for i, sp in enumerate(self.shape_list):
            recover.append(params[self.itx[i]: self.itx[i+1]].reshape(sp))
        params = tuple(p.cuda() for p in recover)
        
        self.model = load_weights(self.model, self.names, params)
        self.model.to(device=x.device)
        out = self.model(x)
        return out
    
    def get_model(self, logger=None, rank=0, add_decay=True):
        mask = self.compute_mask(eval=True)
        if logger is not None and rank == 0:
            logger.log(mask)
        select_model = []
        steps = []
        for i, logit in enumerate(mask):
            if logit == 1:
                select_model.append(self.model_list[i])
                steps.append(i)
        self.infer_model = load_and_average_models(self.infer_model, select_model, steps, self.model_list[-1], add_decay)
        return mask

    def get_action(self, x):
        self.infer_model.to(device=x.device)
        out = self.infer_model(x)
        return out