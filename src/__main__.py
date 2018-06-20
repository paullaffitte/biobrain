import sys

import example
import biobrain

if __name__ == '__main__':
    try:
        if len(sys.argv) != 3:
            print ('Usage: python3 src learn|load filename')
        elif sys.argv[1] == 'learn':
            example.learnigExample(sys.argv[2])
        elif sys.argv[1] == 'load':
            example.loadingExample(sys.argv[2])
    except biobrain.BiobrainException as e:
        print(str(e.__class__.__name__) + ': ' + e.error, file=sys.stderr)