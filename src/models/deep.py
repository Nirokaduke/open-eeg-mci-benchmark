import argparse
def main(cv='loso', epochs=40):
    print('Deep learning baseline (stub).')
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv', default='loso')
    ap.add_argument('--epochs', type=int, default=40)
    args = ap.parse_args()
    main(args.cv, args.epochs)
