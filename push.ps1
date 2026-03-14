Set-Location "D:\datascience"
git add -A
git commit -m "Moved capstone 3&4 from scenario-based to capstone folder in week-02"
git push origin main
Remove-Item -Force "D:\datascience\push.ps1"
Write-Host "Done"
