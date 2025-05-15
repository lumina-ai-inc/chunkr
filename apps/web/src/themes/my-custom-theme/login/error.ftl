<#import "template.ftl" as layout>
<@layout.registrationLayout displayMessage=true; section>
    <#if section = "form">
        <div class="container">
            <div class="login-left">
                <div class="logo">Chunkr</div>
                <div class="login-content">
                    <h1>Error</h1>
                    <div class="alert alert-error">
                        <p>An unexpected error occurred. Please try again or contact support.</p>
                        <p>Error ID: ${errorId!'Unknown'}</p>
                    </div>
                    <a class="register-link" href="${url.loginUrl}">Return to Login</a>
                </div>
            </div>
            <div class="login-right"></div>
        </div>
    </#if>
</@layout.registrationLayout>