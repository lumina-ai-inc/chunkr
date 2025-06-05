<#import "template.ftl" as layout>
<@layout.registrationLayout; section>
    <#if section = "header">
        <h1>Verify Email</h1>
    <#elseif section = "form">
        <div class="container">
            <div class="login-left">
                <div class="logo">Chunkr</div>
                <div class="login-content">
                    <h1>Verify Your Email</h1>
                    <p>An email has been sent to ${kcSanitize(user.email)?no_esc}. Please click the link to verify your email address.</p>
                    <form id="kc-verify-email-form" action="${url.loginAction}" method="post">
                        <button type="submit">Resend Email</button>
                    </form>
                    <a href="${url.loginUrl}" class="register-link">Back to Login</a>
                </div>
            </div>
            <div class="login-right">
             <img src="${url.resourcesPath}/img/code_window_final.webp" alt="Code Window" style="width: 100%; height: 100vh; object-fit: cover;">
            </div>
        </div>
    </#if>
</@layout.registrationLayout>