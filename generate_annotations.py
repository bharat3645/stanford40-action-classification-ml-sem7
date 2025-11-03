"""
Generate annotation files for Stanford 40 dataset from filenames
Since we have images with naming pattern: action_XXX.jpg
"""

import os
import xml.etree.ElementTree as ET
import config

def create_simple_annotations():
    """Create simple XML annotations from image filenames"""
    
    images_dir = config.IMAGES_DIR
    annotations_dir = config.ANNOTATIONS_DIR
    
    # Create annotations directory
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Get all images
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    print(f"Found {len(image_files)} images")
    print(f"Generating annotations in: {annotations_dir}")
    
    created = 0
    for image_file in image_files:
        # Extract action name from filename (e.g., "applauding_001.jpg" -> "applauding")
        action_name = '_'.join(image_file.split('_')[:-1])
        
        # Create XML annotation
        annotation = ET.Element('annotation')
        
        # Add filename
        filename_elem = ET.SubElement(annotation, 'filename')
        filename_elem.text = image_file
        
        # Add folder
        folder_elem = ET.SubElement(annotation, 'folder')
        folder_elem.text = 'JPEGImages'
        
        # Add action
        action_elem = ET.SubElement(annotation, 'action')
        action_elem.text = action_name
        
        # Add object (simplified - no bounding box)
        object_elem = ET.SubElement(annotation, 'object')
        name_elem = ET.SubElement(object_elem, 'name')
        name_elem.text = action_name
        
        # Save XML file
        xml_filename = image_file.replace('.jpg', '.xml')
        xml_path = os.path.join(annotations_dir, xml_filename)
        
        tree = ET.ElementTree(annotation)
        tree.write(xml_path)
        created += 1
    
    print(f"Created {created} annotation files")
    return created

if __name__ == "__main__":
    create_simple_annotations()
