'''import os
import sys

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms

sys.path.append(os.path.abspath('./Matcher'))
from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils

print("MATCHER 로컬 모듈 로드 완료. 모델 생성 중...")

dinov2_kwargs = dict(
    img_size=518,
    patch_size=14,
    init_values=1e-5,
    ffn_layer='mlp',
    block_chunks=0,
    qkv_bias=True,
    proj_bias=True,
    ffn_bias=True,
)

model = vits.__dict__['vit_large'](**dinov2_kwargs)

weights_path = 'model/dinov2_vitl14_pretrain.pth'
dinov2_utils.load_pretrained_weights(model, weights_path, 'teacher')

model = model.cuda().eval()
print("GPU 세팅 끝")

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((518,518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def create_anchor_feature(image_path, mask_path, model, save_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 없음: {image_path}")
        
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    image_tensor = preprocess(image).unsqueeze(0).cuda()

    with torch.no_grad(): # 로컬 모듈은 구형 환경에서도 여기서 터지지 않음
        # MATCHER는 x_norm_patchtokens 대신 x_prenorm을 주로 사용함
        feature = model.forward_features(image_tensor)["x_prenorm"][:, 1:]

        c = feature.shape[-1]
        h = w = int(np.sqrt(feature.shape[1]))
        features = feature.permute(0, 2, 1).reshape(1, c, h, w)

    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float().cuda()
    mask_resized = F.interpolate(mask_tensor, size=(h,w), mode='nearest')
    mask_resized = mask_resized / 255.0
    
    masked_feature = features * mask_resized
    tumor_vector = masked_feature.sum(dim=(2,3)) / (mask_resized.sum(dim=(2,3)) + 1e-6)

    print(f"Extracted feature shape: {tumor_vector.shape}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(tumor_vector.cpu(), save_path)
    print(f"Saved anchor feature to {save_path}")


create_anchor_feature(
    image_path='../BUSI_test_set/image/malignant (206).png', 
    mask_path='../BUSI_test_set/groundtruth/malignant (206)_mask.png', 
    model=model, 
    save_path='save_image/anchor_malignant.pt'
)'''

import os
import sys
import glob
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms

sys.path.append(os.path.abspath('./Matcher'))
from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils

def build_model(device):
    model = vits.__dict__["vit_large"](patch_size=14, img_size=518, block_chunks=0, init_values=1.0)
    model.to(device)
    model.eval()

    weights_path = 'model/dinov2_vitl14_pretrain.pth'
    dinov2_utils.load_pretrained_weights(model, weights_path, 'teacher')
    return model

def extract_centroid(model, img_dir, mask_dir, save_path, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((518,518), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ])

    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))

    if len(img_paths) == 0:
        print(f"[{img_dir}] 폴더에 이미지 X")
        return

    accumulated_features =[]

    with torch.no_grad():
        for img_path in img_paths:
            basename = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, basename)

            if not os.path.exists(mask_path):
                name, ext = os.path.splitext(basename)
                mask_path = os.path.join(mask_dir, f"{name}_mask{ext}")

            if not os.path.exists(mask_path):
                print(f"마스크 누락 패스: {basename}")
                continue

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            mask = cv2.resize(mask, (37, 37), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.tensor(mask, device=device) > 127

            img_tensor = transform(img).unsqueeze(0).to(device)

            features = model.forward_features(img_tensor)['x_norm_patchtokens']
            features = features.squeeze(0)

            mask_flat = mask_tensor.flatten()
            if mask_flat.sum() == 0:
                continue

            fg_features = features[mask_flat]
            img_centroid = fg_features.mean(dim=0)

            accumulated_features.append(img_centroid)
            print(f"처리 완료: {basename}")

    if len(accumulated_features) > 0:
        all_feats_tensor = torch.stack(accumulated_features)
        final_centroid = all_feats_tensor.mean(dim=0)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(final_centroid, save_path)
        print(f"대푯값 앵커 저장 완 {len(accumulated_features)}장 평균 : save_path")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('모델 로드 중...')
    model = build_model(device)

    print("\n--- 양성(Benign) 대푯값 추출 ---")
    extract_centroid(model, 
                     'centroid_data/benign/images', 
                     'centroid_data/benign/masks', 
                     'save_image/anchor_benign_centroid.pt', 
                     device)

    print("\n--- 악성(Malignant) 대푯값 추출 ---")
    extract_centroid(model, 
                     'centroid_data/malignant/images', 
                     'centroid_data/malignant/masks', 
                     'save_image/anchor_malignant_centroid.pt', 
                     device)

if __name__ == '__main__':
    main()