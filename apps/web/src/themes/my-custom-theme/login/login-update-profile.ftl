<#import "template.ftl" as layout>
<@layout.registrationLayout displayRequiredFields=true; section>
    <#if section = "header">
        <h1>Update Profile</h1>
    <#elseif section = "form">
        <div class="container">
            <div class="login-left">
                <div class="logo">Chunkr</div>
                <div class="login-content">
                    <h1>Update Your Profile</h1>
                    <#if message?has_content>
                        <div class="alert ${message.type}">
                            ${kcSanitize(message.summary)?no_esc}
                        </div>
                    </#if>
                    <form id="kc-update-profile-form" action="${url.loginAction}" method="post">
                        <div class="input-group">
                            <label for="username">Username</label>
                            <input type="text" id="username" name="username" value="${(user.username!'')}" readonly />
                        </div>
                        <div class="input-group">
                            <label for="email">Email</label>
                            <input type="email" id="email" name="email" value="${(user.email!'')}" required />
                        </div>
                        <button type="submit">Submit</button>
                        <a href="${url.loginUrl}" class="register-link">Back to Login</a>
                    </form>
                </div>
            </div>
            <div class="login-right"> <img src="${url.resourcesPath}/img/code_window_final.webp" alt="Code Window" style="width: 100%; height: 100vh; object-fit: cover;"></div>
        </div>
    </#if>
</@layout.registrationLayout>