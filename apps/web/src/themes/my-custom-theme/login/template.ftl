<#macro registrationLayout bodyClass="" displayInfo=false displayMessage=false displayRequiredFields=false>
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${msg("loginTitle",(realm.displayName!''))}</title>
    <link rel="stylesheet" href="${url.resourcesPath}/css/login.css" />
</head>
<body class="${bodyClass}">
    <div class="kc-container">
        <div class="container">
            <#nested "header">
            <#nested "form">
            <#if displayInfo>
                <div class="kc-info">
                    <#nested "info">
                </div>
            </#if>
        </div>
    </div>
</body>
</html>
</#macro>