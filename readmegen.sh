cat readmetemplate.md > README.md
echo "
" >> README.md
fd . --type d --maxdepth 2 | sed "s/^/- /" >> README.md
