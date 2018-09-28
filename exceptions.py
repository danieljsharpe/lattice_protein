#!/usr/bin/env python

import sys
import ext_throws


def throws():
    raise ValueError('this is the error message')

def main():
    try:
#        throws()
        ext_throws.ext_throws()
        return 0
    except Exception, err:
        sys.stderr.write('ERROR: %s\n' % str(err))
        return 1

if __name__ == '__main__':
    sys.exit(main())
