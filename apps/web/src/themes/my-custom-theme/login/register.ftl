<#import "template.ftl" as layout>
<@layout.registrationLayout displayRequiredFields=true displayMessage=true; section>
    <#if section = "form">
        <div class="container">
            <div class="login-left">
                <div class="logo">Chunkr</div>
                <div class="login-content">
                    <h1>Create an Account</h1>
                    <#if social.providers??>
                        <div class="social-buttons">
                            <#list social.providers as p>
                                <a href="${p.loginUrl}" class="social-btn ${p.alias}">
                                    <img src="${url.resourcesPath}/img/${p.alias}.ico" alt="${p.displayName}">
                                    Sign up with ${p.displayName}
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
                    <form id="kc-register-form" action="${url.registrationAction}" method="post">
                        <div class="input-group">
                            <label for="username">Username</label>
                            <input type="text" id="username" name="username" value="${(register.formData.username!'')}" required />
                        </div>
                        <div class="input-group">
                            <label for="email">Email</label>
                            <input type="email" id="email" name="email" value="${(register.formData.email!'')}" required />
                        </div>
                        <div class="input-group">
                            <label for="password">Password</label>
                            <input type="password" id="password" name="password" required />
                        </div>
                        <div class="input-group">
                            <label for="password-confirm">Confirm Password</label>
                            <input type="password" id="password-confirm" name="password-confirm" required />
                        </div>
                        <button type="submit">Register</button>
                        <a href="${url.loginUrl}" class="register-link">Already have an account? Sign In</a>
                    </form>
                </div>
            </div>
            <div class="login-right">
              <img src="${url.resourcesPath}/img/code_window_final.webp" alt="Code Window" style="width: 100%; height: 100vh; object-fit: cover;">
            </div>
        </div>
    </#if>
</@layout.registrationLayout>