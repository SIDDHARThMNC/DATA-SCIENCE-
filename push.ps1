Set-Location "D:\datascience"
git add -A
git commit -m "Restructured: organized all files into week-01-data-science, week-02-ai-ml, week-03-ai-ml-advanced-prompt, extra-curricular-activity"
git push origin main
Remove-Item -Force "D:\datascience\push.ps1"
Write-Host "Pushed successfully!"
