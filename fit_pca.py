import os
import glob
import torch
import cv2
from torchvision import transforms
import sys

sys.path.append(os.path.abspath('./Matcher'))
from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils

def build_model(device):
    model = vits.__dict__["vit_large"](patch_size=14, img_size=518, block_chunks=0, init_values=1.0)
    model.to(device)
    model.eval()
    weights_path = '/model/dinov2_vitl14_pretrain.pth'
    dinov2_utils.load_pretrained_weight(model, weights_path, 'teacher')
    return model

def get_all_features(model, base_dir, device):
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((518, 518), antialias=True),
        transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
    ])

    all_features = []

    for cls in ['benign', 'malignant']:
        img_dir = os.path.join(base_dir, cls, 'images')
        mask_dir = os.path.join(base_dir, cls, 'masks')

        img_paths = glob.glob(os.path.join(img_dir, ".png"))

        with torch.no_grad():
            for img_path in img_paths:
                basename = os.path.basename(img_path)
                name, ext = os.path.splitext(basename)

                mask_path = os.path.join(mask_dir, basename)
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(mask_dir, f"{name}_mask.png")
                if not os.path.exists(mask_path):
                    continue

                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (37,37), interpolation=cv2.INTER_NEAREST)
                mask_tensor = torch.tensor(mask, device= device) > 127

                img_tensor = transforms(img).unsqueeze(0).to(device)
                features = model.forward_features(img_tensor)['x_norm_patchtokens'].squeeze(0)

                fg_features = features[mask_tensor.flatten()]
                all_features.append(fg_features)
                print(f"특징 수집 완: {basename}")

    return torch.cat(all_features, dim=0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(device)
    
    print("특징 벡터 수집 중...")
    # 데이터 폴더 경로는 실제 환경에 맞게 수정
    features = get_all_features(model, 'centroid_data', device) 
    
    print(f"수집된 총 벡터 크기: {features.shape}")
    print("PCA 학습 중 (1024차원 -> 128차원으로 압축)...")
    
    # 1. 평균(Mean) 계산 및 중심화(Centering)
    pca_mean = features.mean(dim=0, keepdim=True)
    centered_features = features - pca_mean
    
    # 2. PyTorch 내장 PCA (차원 축소)
    # q=128 은 남길 핵심 차원의 수 (이 숫자를 조절하며 최적화 가능)
    U, S, V = torch.pca_lowrank(centered_features, q=128)
    
    # 3. 평균 벡터와 투영 행렬(V) 저장
    os.makedirs('save_image', exist_ok=True)
    torch.save(pca_mean, 'save_image/pca_mean.pt')
    torch.save(V, 'save_image/pca_V.pt') # V 행렬의 형태는 [1024, 128]
    print("-> PCA 변환 행렬 저장 완료! (save_image/pca_mean.pt, pca_V.pt)")

if __name__ == '__main__':
    main()
