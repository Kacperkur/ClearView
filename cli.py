# Analyze a single image
## python cli.py detect --input photo.jpg

# Analyize images from an article using URL ( scraped from the website)
## python cli.py detect --url https://www.example.come/article/image.jpg

# analyze images from a video frame or frames
## python cli.py detect --video path/to/video.mp4 --frame 100-200

# Batch process a folder of images
## python cli.py detect --batch /path/to/folder 

# generate a GradCAM heatmap to visualize what part of the image is being focused on by the model
## python cli.py gradcam --input photo.jpg --output heatmap.jpg


import torch
print(torch.__version__)              # Should show 2.x+cu128
print(torch.cuda.is_available())      # Should be True
print(torch.cuda.get_device_name(0))  # Should show RTX 5060 Ti
print(torch.cuda.get_device_capability(0))  # Should show (12, 0)
