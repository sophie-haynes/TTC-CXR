from collections import Counter
import os
import os.path
import random
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset


class CXRNoduleData(VisionDataset):
    """
    CXR Nodule dataset that loads images from folder structure.
    Expected folder structure:
    img_root/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── class2/
        ├── image3.jpg
        ├── image4.jpg
        └── ...
    """

    def __init__(
        self,
        img_root: str = "",
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        minority_class: str = "normal",
        classes: Optional[List[str]] = None,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        # test_ratio will be 1 - train_ratio - val_ratio
        skip_split = True
    ) -> None:

        self.split = split
        self.minority_class = minority_class
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.skip_split = skip_split
        
        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
        if self.test_ratio < 0:
            raise ValueError("train_ratio + val_ratio must be <= 1.0")

        super().__init__(img_root, transform=transform, target_transform=target_transform)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. Check the root path.")

        # Get all class directories
        self.all_classes: List[str] = []
        try:
            self.all_classes = sorted([d for d in os.listdir(self.root) 
                                     if os.path.isdir(os.path.join(self.root, d))])
        except FileNotFoundError:
            raise RuntimeError(f"Directory not found: {self.root}")

        if not self.all_classes:
            raise RuntimeError(f"No class directories found in: {self.root}")

        # Use specified classes or all available classes
        if classes is None:
            self.classes = self.all_classes
            print(f"Using all available classes: {self.classes}")
        else:
            # Validate that specified classes exist
            missing_classes = set(classes) - set(self.all_classes)
            if missing_classes:
                raise ValueError(f"Classes not found in dataset: {missing_classes}")
            self.classes = classes
            print(f"Using specified classes: {self.classes}")

        # Create class name to ID mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Index of all files: (class_id, class_name, filepath)
        self.index: List[Tuple[int, str, str]] = []
        
        # Build the full dataset index first
        all_samples = []
        for class_name in self.classes:
            class_id = self.class_to_idx[class_name]
            class_path = os.path.join(self.root, class_name)
            print(f"========= Looking for files in: {class_path}")
            try:
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
                
                for img_file in image_files:
                    img_path = os.path.join(class_path, img_file)
                    all_samples.append((class_id, class_name, img_path))
                    
            except FileNotFoundError:
                print(f"Warning: Class directory not found: {class_path}")

        if not all_samples:
            raise RuntimeError("No valid image files found in any class directory")

        # Shuffle samples with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(all_samples)

        if not self.skip_split:
            # Split data into train/val/test
            n_total = len(all_samples)
            n_train = int(self.train_ratio * n_total)
            n_val = int(self.val_ratio * n_total)
            n_test = n_total - n_train - n_val
    
            train_samples = all_samples[:n_train]
            val_samples = all_samples[n_train:n_train + n_val]
            test_samples = all_samples[n_train + n_val:]
    
            # Select samples for current split
            if self.split == "train":
                self.index = train_samples
            elif self.split == "val":
                self.index = val_samples
            elif self.split == "test":
                self.index = test_samples
        else:
            self.index = all_samples

        self._print_dataset_info()

    def _print_dataset_info(self) -> None:
        """Print dataset information and class statistics."""
        print(f'Created dataset {self.__class__.__name__} with {len(self)} samples.')
        print(f'Split: {self.split}')
        print(f'Classes: {self.classes}')
        print(f'Minority class: {self.minority_class}')
        
        # Count samples per class
        cls_counter = Counter(cls_id for (cls_id, _, _) in self.index)
        cls_counts = [cls_counter.get(i, 0) for i in range(len(self.classes))]
        print(f'Class counts: {dict(zip(self.classes, cls_counts))}')
        
        # Calculate class ratios
        total_samples = len(self.index)
        if total_samples > 0:
            cls_ratios = [count / total_samples for count in cls_counts]
            print(f'Class ratios: {dict(zip(self.classes, cls_ratios))}')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get item at the specified index.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, target) where target is the class index
        """
        class_id, class_name, img_path = self.index[index]
        
        try:
            # Load image and convert to RGB
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        target = class_id

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.index)

    def _check_integrity(self) -> bool:
        """Check if the dataset directory exists and is not empty."""
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0

    def get_class_weights(self) -> List[float]:
        """
        Calculate class weights for handling imbalanced datasets.
        Returns inverse frequency weights.
        """
        cls_counter = Counter(cls_id for (cls_id, _, _) in self.index)
        total_samples = len(self.index)
        
        weights = []
        for i in range(len(self.classes)):
            class_count = cls_counter.get(i, 1)  # Avoid division by zero
            weight = total_samples / (len(self.classes) * class_count)
            weights.append(weight)
        
        return weights