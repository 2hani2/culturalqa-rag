"""Matched research training replaces the legacy unsplit training script."""
from research_train import train
if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--max_examples",type=int,default=60);p.add_argument("--epochs",type=int,default=8);p.add_argument("--text_only",action="store_true");p.add_argument("--enable_images",action="store_true",help="Images are enabled by default for matched research training.");a=p.parse_args();train(train_per_dataset=a.max_examples,epochs=a.epochs,enable_images=not a.text_only)
