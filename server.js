const express = require('express');
const http = require('http');
const path = require('path');
const fs = require('fs');
const jimp = require('jimp');
const bodyParser = require('body-parser');
const {spawn} = require('child_process');
const ejs = require('ejs');
const jsonfile = require('jsonfile');
const multer = require('multer');
require('./public/App.test.js');
require('dotenv').config();

const app = express();
let server = http.createServer(app);
const PORT = process.env.PORT || 9000;
const AppName = "Catofication";

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

app.use('/images',express.static(path.join(__dirname,'images')));
app.use('/public',express.static(path.join(__dirname,'public')));
app.use('/weight',express.static(path.join(__dirname,'weight')));

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

const storage = multer.memoryStorage();
const upload = multer({storage: storage});

const promises = [
    ejs.renderFile('./views/header.ejs')
];

app.get('/', (req, res) => {
    Promise.all(promises).then(([header, feed]) => {
        res.status(200).render('index',{header, feed});
    });
});

app.post('/process', upload.single('file'), async (req, res) => {
    try{
        const extension = "png";
        const tempFilePath = path.join(__dirname,'/weight/bin/temp_image.png');
        const fileBuffer = req.file.buffer;
        const image = await jimp.read(fileBuffer);
        await image.writeAsync(`${tempFilePath}`);
        const listOfInput = [tempFilePath.replaceAll('\\','/'), extension];
        await callPythonProcess(listOfInput, 'logistic').then(output => {
            res.status(200).json({output});
        }).catch(error => {
            console.error('Error:', error);
        });
    }catch(e){
        res.status(403).render('notfound',{error: 403, message: "Failed to process most recent task, Try again later"});
    }
});

function callPythonProcess(list, functionValue){
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', ['./model/main.py', list, functionValue]);
        let resultData = '';
        pythonProcess.stdout.on('data', (data) => {
            resultData += data.toString();
        });
        pythonProcess.stderr.on('data', (data) => {
            console.error(`stderr: ${data}`);
        });
        pythonProcess.on('close', (code) => {
            if(code !== 0){
                console.log(`Python script exited with code ${code}`);
            }
            try{
                const result = JSON.parse(resultData);
                resolve(result);
            }catch(error){
                console.error(`Error parsing JSON: ${error.message}`);
                reject(new Error("Error parsing JSON from Python script"));
            }
        });
    });
}

app.get('*', (req, res) => {
    res.status(404).render('notfound',{error: 404, message: "Page not found on this url, check the source or report it"});
});

server.listen(PORT, (err) => {
    if(err) console.log("Oops an error occure:  "+err);
    console.log(`Compiled successfully!\n\nYou can now view \x1b[33m./${path.basename(__filename)}\x1b[0m in the browser.`);
    console.info(`\thttp://localhost:${PORT}`);
    console.log("\n\x1b[32mNode web compiled!\x1b[0m \n");
});
