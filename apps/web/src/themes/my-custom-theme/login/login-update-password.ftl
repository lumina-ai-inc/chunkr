<#import "template.ftl" as layout>
<@layout.registrationLayout displayRequiredFields=true; section>
    <#if section = "header">
        <h1>Change Password</h1>
    <#elseif section = "form">
        <div class="login-left">
            <div class="logo">Chunkr</div>
            <div class="login-content">
                <h1>Set New Password</h1>
                <p>Enter and confirm your new password.</p>
                <#if message?has_content>
                    <div class="alert ${message.type}">
                        ${kcSanitize(message.summary)?no_esc}
                    </div>
                </#if>
                <form id="kc-passwd-update-form" action="${url.loginAction}" method="post">
                    <input type="hidden" name="username" value="${(resetPassword.username!'')}" />
                    <input type="hidden" name="email" value="${(resetPassword.email!'')}" />
                    <div class="input-group">
                        <input type="password" id="password-new" name="password-new" placeholder="New Password" autocomplete="new-password" required />
                    </div>
                    <div class="input-group">
                        <input type="password" id="password-confirm" name="password-confirm" placeholder="Confirm Password" required />
                    </div>
                    <button type="submit">Change Password</button>
                    <a href="${url.loginUrl}" class="register-link">Back to Login</a>
                </form>
            </div>
        </div>
        <div class="login-right"> <img src="${url.resourcesPath}/img/code_window_final.webp" alt="Code Window" style="width: 100%; height: 100vh; object-fit: cover;"></div>
    </#if>
</@layout.registrationLayout>