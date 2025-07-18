from PIL import Image, ImageDraw, ImageFont
import os


def merge_images():
    # 디렉토리 경로 설정
    base_dirs = {
        "sparse": "sparse",
        "dense": "dense",
        "s2d": "s2d",
        "d2s": "d2s",
    }

    # 스텝 목록 (이미지 파일 이름에서 추출)
    steps = [f"{i}.png" for i in range(0, 10000001, 1000000)]

    # 라벨용 폰트 설정 (시스템에 따라 조절)
    try:
        font = ImageFont.truetype("arial.ttf", size=64)
    except:
        font = ImageFont.load_default(size=64)

    output_dir = "combined_images"
    os.makedirs(output_dir, exist_ok=True)

    for step in steps:
        # 2x2 원본 이미지 배열 만들기
        images = {}
        for key, path in base_dirs.items():
            img_path = os.path.join("trajectory_images", path, step)
            if os.path.exists(img_path):
                images[key] = Image.open(img_path).convert("RGBA")
            else:
                # 없을 경우 흰색 placeholder
                images[key] = Image.new("RGBA", (700, 1000), (255, 255, 255, 255))

        # 최종 캔버스 생성 (라벨 포함 공간: 2000x2000)
        canvas = Image.new("RGBA", (2000, 2000), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # 위치 정의 및 삽입
        canvas.paste(images["sparse"], (300, 0))
        canvas.paste(images["dense"], (1000, 0))
        canvas.paste(images["s2d"], (300, 1000))
        canvas.paste(images["d2s"], (1000, 1000))

        # 라벨 텍스트 추가
        draw.text((50, 500), "sparse", fill="black", font=font)
        draw.text((50, 1500), "s2d", fill="black", font=font)
        draw.text((1800, 500), "dense", fill="black", font=font, anchor="ra")
        draw.text((1800, 1500), "d2s", fill="black", font=font, anchor="ra")

        # 스텝 정보도 상단 중앙에 추가
        draw.text(
            (1000, 30),
            f"Step {step.replace('.png', '')}",
            fill="black",
            font=font,
            anchor="mm",
        )

        # 저장
        output_path = os.path.join(output_dir, f"combined_{step}")
        canvas.save(output_path)

    print("✅ 이미지 병합 완료")


if __name__ == "__main__":
    merge_images()
