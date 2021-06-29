# posh-git helps git auto-complete and branch recognition
Import-Module posh-git

# change the color of the pathname to an organge color
$GitPromptSettings.DefaultPromptPath.ForegroundColor = 0xFFA500
# start the user input on a newline from the prompt
$GitPromptSettings.DefaultPromptBeforeSuffix.Text = '`n'

# posh-docker helps auto-complete docker commands
Import-Module posh-docker

# remove the "bad" ls command and replace it with a better one
Remove-Alias -Name ls
function ls {
  Get-ChildItem | Format-Wide -Column 4
}

# func for listing docker containers a little more cleanly
function dps {
  docker ps -a --format "table {{.ID}}\t{{.Names}}\t{{.Image}}"
}

# vim command that uses bash vim instead
function vim {
  param(
    [Parameter(Mandatory=$true)]
    [String]
    $filename
  )
  
  bash -c "vim $filename"
}

# nice variable for my notes directory
$notes = "C:\dev\notes"
