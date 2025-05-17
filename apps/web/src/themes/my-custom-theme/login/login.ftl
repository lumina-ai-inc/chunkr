<#import "template.ftl" as layout>
<@layout.registrationLayout displayRequiredFields=false displayMessage=true; section>
    <#if section = "form">
        <div class="container">
            <div class="login-left">
                <div class="logo">Chunkr</div>
                <div class="login-content">
                    <h1>Welcome!</h1>
                    <#if social.providers??>
                        <div class="social-buttons">
                            <#list social.providers as p>
                                <a href="${p.loginUrl}" class="social-btn ${p.alias}">
                                    <img src="${url.resourcesPath}/img/${p.alias}.ico" alt="${p.displayName}">
                                    Sign in with ${p.displayName}
                                </a>
                            </#list>
                        </div>
                    </#if>
                    <div class="or-divider">
                        <hr>
                        <span>OR</span>
                        <hr>
                    </div>
                    <#if message?has_content>
                        <div class="alert alert-${message.type}">
                            ${kcSanitize(message.summary)?no_esc}
                        </div>
                    </#if>
                    <form id="kc-form-login" action="${url.loginAction}" method="post">
                        <div class="input-group">
                            <label for="username">Email or Username</label>
                            <input id="username" name="username" type="text" value="${(login.username!'')}" required />
                        </div>
                        <div class="input-group">
                            <label for="password">Password</label>
                            <input id="password" name="password" type="password" required />
                            <#if realm.resetPasswordAllowed>
                                <a class="forgot-password" href="${url.loginResetCredentialsUrl}">Forgot Password?</a>
                            </#if>
                        </div>
                        <button type="submit">Sign In</button>
                        <#if realm.registrationAllowed>
                            <a class="register-link" href="${url.registrationUrl}">Register</a>
                        </#if>
                    </form>
                </div>
            </div>
           <div class="login-right">
    <img src="${url.resourcesPath}/img/code_window_final.webp" alt="Code Window" style="width: 100%; height: 100vh; object-fit: cover;">
</div>
        </div>
    </#if>
</@layout.registrationLayout>