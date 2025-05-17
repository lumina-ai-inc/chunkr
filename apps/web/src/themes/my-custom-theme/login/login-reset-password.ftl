<#import "template.ftl" as layout>
<@layout.registrationLayout displayRequiredFields=false; section>
    <#if section = "header">
      
    <#elseif section = "form">
        <div class="container">
            <div class="login-left">
          
                <div class="login-content">
                    <h1>Forgot Your Password?</h1>
                    <p>Enter your email or username to receive a password reset link.</p>
                    <#if message?has_content>
                        <div class="alert alert-${message.type}">
                            ${kcSanitize(message.summary)?no_esc}
                        </div>
                    </#if>
                    <form id="kc-reset-password-form" action="${url.loginAction}" method="post">
                        <div class="input-group">
                            <input type="text" id="username" name="username" placeholder="Email or Username" value="${(auth.attemptedUsername!'')}" required />
                        </div>
                        <button type="submit">Send Reset Link</button>
                        <a href="${url.loginUrl}" class="register-link">Back to Login</a>
                    </form>
                </div>
            </div>
            <div class="login-right">
                <img src="${url.resourcesPath}/img/code_window_final.webp" alt="Code Window" style="width: 100%; height: 100vh; object-fit: cover;">
            </div>
        </div>
    </#if>
</@layout.registrationLayout>