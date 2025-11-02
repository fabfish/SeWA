import paddle
import numpy as np
from src.vit import VisionTransformer
import matplotlib.pyplot as plt

def test_vit_model():
    """测试Vision Transformer模型是否能正常运行"""
    print("开始测试Vision Transformer模型...")
    
    # 创建一个小型ViT模型用于测试
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=100,
        embed_dim=192,
        depth=4,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1
    )
    
    # 打印模型结构
    print(f"模型结构:\n{model}")
    
    # 创建随机输入数据
    batch_size = 4
    x = paddle.randn([batch_size, 3, 32, 32])
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")
    print(f"输出类型: {type(output)}")
    
    # 检查模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params}")
    
    return True

def visualize_random_image():
    """生成并显示一个随机图像"""
    # 创建随机图像
    random_img = np.random.rand(32, 32, 3)
    
    # 显示图像
    plt.figure(figsize=(3, 3))
    plt.imshow(random_img)
    plt.title("随机测试图像")
    plt.axis('off')
    plt.savefig("random_test_image.png")
    print("已保存随机测试图像到 random_test_image.png")

if __name__ == "__main__":
    print("PaddlePaddle版本:", paddle.__version__)
    
    # 测试ViT模型
    test_success = test_vit_model()
    
    if test_success:
        print("\n✅ 模型测试成功！")
        # 生成测试图像
        try:
            visualize_random_image()
        except Exception as e:
            print(f"生成测试图像时出错: {e}")
    else:
        print("\n❌ 模型测试失败！")