# Usage example
if __name__ == "__main__":
    detector = PhoneDetector(model_path='best_model_state_dict_f12.pth', img_height=150, img_width=150, capture_interval=20)
    detector.run()