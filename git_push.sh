git status
git add ./code
git add ./test.ipynb
git add ./wbw_test.py
currentdate=$(date +%Y%m%d)
git commit -m $currentdate
git push -f origin main