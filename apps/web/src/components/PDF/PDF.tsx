import { useState, useEffect, useId } from "react";
import { Document, Page } from 'react-pdf';
import * as pdfjs from 'pdfjs-dist';

function REACTPDF() {
    const [numPages, setNumPages] = useState<number>();
    const [pageNumber, setPageNumber] = useState<number>(1);

    function onDocumentLoadSuccess({ numPages }: { numPages: number }): void {
        setNumPages(numPages);
    }

    return (
        <div>
            <Document file="https://raw.githubusercontent.com/mozilla/pdf.js/ba2edeae/examples/learning/helloworld.pdf" onLoadSuccess={onDocumentLoadSuccess}>
                <Page pageNumber={pageNumber} />
            </Document>
            <p>
                Page {pageNumber} of {numPages}
            </p>
        </div>
    );
}

export const PDF = () => {
    // const url = 'https://raw.githubusercontent.com/mozilla/pdf.js/ba2edeae/examples/learning/helloworld.pdf'
    const id = useId();
    //
    // useEffect(() => {
    //
    //     initDocument()
    //         .catch(console.error)
    // }, []);
    //
    // const initDocument = async () => {
    //     console.log("prepdf");
    //
    //     pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;
    //
    //     const pdf = await pdfjs.getDocument(url).promise;
    //
    //     const pageNumber = 1;
    //     const page = await pdf.getPage(pageNumber);
    //
    //     const scale = 1.5;
    //     const viewport = page.getViewport({ scale: scale });
    //
    //     // Prepare canvas using PDF page dimensions
    //     const canvas: HTMLCanvasElement = document.getElementById(id) as HTMLCanvasElement;
    //     const context = canvas.getContext('2d')!;
    //     canvas.height = viewport.height;
    //     canvas.width = viewport.width;
    //
    //     // Render PDF page into canvas context
    //     const renderContext = {
    //         canvasContext: context,
    //         viewport: viewport
    //     };
    //     const renderTask = page.render(renderContext);
    //     await renderTask.promise;
    //     //
    //     // canvas.width = Math.floor(viewport.width);
    //     // canvas.height = Math.floor(viewport.height);
    //     // canvas.style.width = Math.floor(viewport.width) + "px";
    //     // canvas.style.height = Math.floor(viewport.height) + "px";
    //     // console.log("pdf");
    // };

    const showA = false;

    return (
        <div>
            {showA &&
                <canvas id={id}>
                </canvas>
            }
            {!showA &&
                <REACTPDF />
            }
        </div>
    );
};

