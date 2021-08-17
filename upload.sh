#!/bin/bash

if [ $# -lt 1 ] ; then
    COMMENT=" updated "
else
    COMMENT=$1
fi

a=$((`awk -F= '{print $2}' version.py  | sed "s/\s*'//g" | awk -F. '{print $NF}'` + 1))
VERSION=`awk -F= '{print $2}' version.py  | sed "s/\s*'//g" | awk -F. 'BEGIN{OFS="."}{print $1,$2,$3}'`
NEW_VERSION=`awk -F= '{print $2}' version.py  | sed "s/\s*'//g" | awk -F. 'BEGIN{OFS="."}{print $1,$2}'`.$a

echo __version__ = \'$VERSION\'
echo __version__ = \'$NEW_VERSION\' > version.py

rm -rf dist
python3 setup.py sdist
python3 setup.py bdist_wheel

git add * -v
git commit -m "$COMMENT"
git push

git tag $VERSION -m $COMMENT
git push --tags

twine check dist/*
twine upload dist/*


