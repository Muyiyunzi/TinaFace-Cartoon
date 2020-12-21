
ls -l data/cartoon/trainset/Annotations | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/cartoon/trainset/train.txt
