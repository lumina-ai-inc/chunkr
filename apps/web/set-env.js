const fs = require('fs');
require('dotenv').config({ path: '.env.local' });

let configFile = fs.readFileSync('./public/config.js', 'utf8');

Object.keys(process.env).forEach(key => {
    const value = process.env[key];
    configFile = configFile.replace(`__${key}__`, value);
});

fs.writeFileSync('./public/config.local.js', configFile);
console.log('Local config generated');