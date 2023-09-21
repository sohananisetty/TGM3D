rm -rf pretrained
mkdir pretrained/extractors
cd pretrained/extractors
echo -e "Downloading extractors"
gdown --fuzzy https://drive.google.com/file/d/1o7RTDQcToJjTm9_mNWTyzvZvjTWpZfug/view
gdown --fuzzy https://drive.google.com/file/d/1KNU8CsMAnxFrwopKBBkC8jEULGLPBHQp/view


unzip t2m.zip
unzip kit.zip

echo -e "Cleaning\n"
rm t2m.zip
rm kit.zip
echo -e "Downloading done!"