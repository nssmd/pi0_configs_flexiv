import cv2
import os
import argparse

def extract_frames(video_path, output_folder, prefix='frame'):
    """
    Extract frames from a video file and save them as images.
    
    Args:
        video_path (str): Path to the video file
        output_folder (str): Path to the folder where frames will be saved
        prefix (str): Prefix for the frame filenames
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count}")
    
    # Read and save frames
    frame_index = 0
    success = True
    
    while success:
        # Read next frame
        success, frame = video.read()
        
        if success:
            # Save frame as image
            frame_filename = f"{prefix}_{frame_index:06d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            # Print progress
            if frame_index % 100 == 0:
                print(f"Extracted frame {frame_index}/{frame_count}")
            
            frame_index += 1
    
    # Release video
    video.release()
    
    print(f"Extraction complete. {frame_index} frames extracted to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("output_folder", help="Path to the output folder")
    parser.add_argument("--prefix", default="frame", help="Prefix for frame filenames")
    
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.output_folder, args.prefix) 