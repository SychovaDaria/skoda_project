if __name__ == "__main__":
    trainer = ModelTrainer(dataset_path='mobil', dataset_path2='ar1', img_height=150, img_width=150, batch_size=32, epochs=30)
    trainer.train()

