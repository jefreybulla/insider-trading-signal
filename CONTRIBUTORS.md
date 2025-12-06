# Contributor procedure
- Create and checkout new working branch
```
git checkout -b <branch-name-here>
```
- Make edits
- Stage your changes
```
git add .
```
- Commit
```
git commit -m "<commit-name-usually-an-action>"
```
- Push branch to GitHub
```
git push origin <branch-name-here>
```
- On GitHub.com click on "Create a new Pull Request"
- Open a new Pull Request from your new branch against then main branch

## Creating/Editing a notebook
For notebooks before opening a Pull Request export your new or edited notebook to a Python file so all contributors can review the changes from Github. 

Export Python file with
```
jupyter nbconvert --to script --ClearOutputPreprocessor.enabled=True your_notebook.ipynb
```


