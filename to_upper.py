# -*- coding: utf-8 -*-
import json

def main():
    with open('labels_vn.json', 'r') as f:
        distros_dict = json.load(f)

    upperCase = [x.upper() for x in distros_dict]
    print(upperCase)
    with open('labels_vn_upper.json', 'w') as outfile:
        json.dump(upperCase, outfile, ensure_ascii=False)


if __name__ == '__main__':
    main()