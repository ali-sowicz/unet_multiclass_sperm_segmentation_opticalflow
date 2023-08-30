import os
from PIL import Image

def combine_and_colorize_images(flagellum_path, head_path, output_path):
    # Create black background image
    width, height = 512, 512
    background = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    
    # Get list of image filenames in the folders
    flagellum_files = sorted(os.listdir(flagellum_path))
    head_files = sorted(os.listdir(head_path))
    
    for flagellum_file, head_file in zip(flagellum_files, head_files):
        flagellum_image = Image.open(os.path.join(flagellum_path, flagellum_file)).convert('L')  # Convert to grayscale
        head_image = Image.open(os.path.join(head_path, head_file)).convert('L')  # Convert to grayscale
        
        # Resize images to match the desired size
        flagellum_image = flagellum_image.resize((width, height), Image.ANTIALIAS)
        head_image = head_image.resize((width, height), Image.ANTIALIAS)
        
        # Create RGBA versions with transparency channels
        red_flagellum = Image.new('RGBA', flagellum_image.size, (255, 0, 0, 0))  # Transparent red
        green_head = Image.new('RGBA', head_image.size, (0, 255, 0, 0))  # Transparent green
        
        # Apply grayscale images as alpha masks to change white pixels to color
        red_flagellum.putalpha(flagellum_image)
        green_head.putalpha(head_image)
        
        # Paste the red flagellum and green head onto the black background
        final_image = Image.alpha_composite(background, red_flagellum)
        final_image = Image.alpha_composite(final_image, green_head)
        
        # Save the resulting image
        output_filename = os.path.splitext(flagellum_file)[0] + ".png"  #"Mask_" + 
        final_image.save(os.path.join(output_path, output_filename))


def convert_jpg_to_png(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            png_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
            img.save(png_path, 'PNG')

def convert_rgba_to_rgb(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            
            # Convert RGBA image to RGB
            img_rgb = img.convert('RGB')
            
            png_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
            img_rgb.save(png_path, 'PNG')

if __name__ == "__main__":
    # Paths to the flagellum and head image folders
    flagellum_folder = '/dih4/dih4_1/summercamp/agolisowicz/slowfast_segmentation/data/train/train_masks_fla'
    head_folder = '/dih4/dih4_1/summercamp/agolisowicz/slowfast_segmentation/data/train/train_masks_head'

    # # Output path for combined images
    output_folder = '/dih4/dih4_1/summercamp/agolisowicz/unet_multiclass/unet-multiclass-pytorch/data/images_train/masks_sperm'

    # # Create the output folder if it doesn't exist
    # os.makedirs(output_folder, exist_ok=True)

    # # Call the function to combine and colorize the images
    combine_and_colorize_images(flagellum_folder, head_folder, output_folder)



    # Paths to the input (JPG) and output (PNG) folders
    input_folder = '/dih4/dih4_1/summercamp/agolisowicz/unet_multiclass/unet-multiclass-pytorch/data/images_train/masks_sperm'
    output_folder = '/dih4/dih4_1/summercamp/agolisowicz/unet_multiclass/unet-multiclass-pytorch/data/images_train/masks_sperm'

    # Call the function to convert JPG images to PNG
    # convert_jpg_to_png(input_folder, output_folder)
    # Call the function to convert RGBA images to RGB
    convert_rgba_to_rgb(input_folder, output_folder)