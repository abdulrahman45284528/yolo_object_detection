import cv2
import torch
from ultralytics import YOLO
import argparse
import os


# Load Pretrained YOLO Model

def load_model(model_name='yolov5s'):
    print(f"[INFO] Loading YOLO model: {model_name}")
    model = YOLO(model_name + '.pt')  # 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt' are available
    return model


# Detect Objects in an Image

def detect_image(model, image_path, output_path):
    print(f"[INFO] Detecting objects in image: {image_path}")
    results = model(image_path)  # Inference
    results.save(filename=output_path)
    print(f"[INFO] Saved detected image to: {output_path}")


# Detect Objects in a Video

def detect_video(model, video_path, output_path):
    print(f"[INFO] Detecting objects in video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection
        results = model(frame)
        annotated_frame = results.render()[0]

        # Write frame to output video
        out.write(annotated_frame)

        # Display live output (Optional)
        cv2.imshow('YOLOv5 Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved detected video to: {output_path}")


# Fine-Tune YOLO on Custom Dataset

def fine_tune(model, data_yaml, epochs):
    print(f"[INFO] Fine-tuning YOLO model for {epochs} epochs on dataset: {data_yaml}")
    model.train(data=data_yaml, epochs=epochs)
    model.save('yolov5_finetuned.pt')
    print("[INFO] Fine-tuning complete and model saved as 'yolov5_finetuned.pt'")


# Main Function

def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection with PyTorch")
    parser.add_argument('--mode', type=str, required=True, choices=['image', 'video', 'fine-tune'])
    parser.add_argument('--input', type=str, help='Path to input image or video')
    parser.add_argument('--output', type=str, help='Path to save the output file')
    parser.add_argument('--model', type=str, default='yolov5s', help='YOLO model to use (e.g., yolov5s, yolov5m)')
    parser.add_argument('--data', type=str, help='Path to dataset YAML file for fine-tuning')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for fine-tuning')

    args = parser.parse_args()

    # Load YOLO model
    model = load_model(args.model)

    if args.mode == 'image':
        if not args.input or not args.output:
            raise ValueError("[ERROR] Please specify both input and output paths for image mode.")
        detect_image(model, args.input, args.output)

    elif args.mode == 'video':
        if not args.input or not args.output:
            raise ValueError("[ERROR] Please specify both input and output paths for video mode.")
        detect_video(model, args.input, args.output)

    elif args.mode == 'fine-tune':
        if not args.data:
            raise ValueError("[ERROR] Please provide path to data.yaml for fine-tuning.")
        fine_tune(model, args.data, args.epochs)

    else:
        raise ValueError("[ERROR] Invalid mode selected.")

if __name__ == "__main__":
    main()
