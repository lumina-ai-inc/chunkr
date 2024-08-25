import { useEffect, useId } from "react";
// import pdfjs from "pdfjs-dist";

export const PDF = () => {
    const url = 'https://raw.githubusercontent.com/mozilla/pdf.js/ba2edeae/examples/learning/helloworld.pdf'
    const id = useId();

    useEffect(() => {

        initDocument()
            .catch(console.error)
    }, []);

    const initDocument = async () => {
        // const pdf = await pdfjs.getDocument(url).promise;
        // const pageNumber = 1;
        // const page = await pdf.getPage(pageNumber);
        //
        // const scale = 1.5;
        // const viewport = page.getViewport({ scale: scale });
        //
        // // Prepare canvas using PDF page dimensions
        // const canvas: HTMLCanvasElement = document.getElementById(id) as HTMLCanvasElement;
        // const context = canvas.getContext('2d')!;
        // canvas.height = viewport.height;
        // canvas.width = viewport.width;
        //
        // // Render PDF page into canvas context
        // const renderContext = {
        //     canvasContext: context,
        //     viewport: viewport
        // };
        // const renderTask = page.render(renderContext);
        // await renderTask.promise;
        //
        // canvas.width = Math.floor(viewport.width);
        // canvas.height = Math.floor(viewport.height);
        // canvas.style.width = Math.floor(viewport.width) + "px";
        // canvas.style.height = Math.floor(viewport.height) + "px";
    };

    return (
        <canvas id={id}>
        </canvas>
    );
};
