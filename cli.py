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