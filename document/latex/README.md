# LaTeX Documents

This folder contains LaTeX documents for the document that the author wrote for
their master thesis. The documents are organized into folders for each project.

## Install

Only `latexmk` is required.

To install latexmk, there are several ways to do it depending on the OS.

On Linux:
It should be already installed. You may have to install a package called
latexmk or similar.

On macOS:
It's probably already installed. If not, open “TeX Live Utility”, search for
“latexmk” and install it. If you prefer using the Terminal:
`sudo tlmgr install latexmk`

On Windows:
You probably have to install MikTeX, e.g. from here:
https://miktex.org/download. If it’s not installed already, open the MikTeX
Package Manager and install the latexmk package.

Alternatively, you can use the TeX Live package manager (tlmgr) to install
latexmk. For example, if you installed vanilla TeX live from TUG, you can
install latexmk via tlmgr
(tlmgr install latexmk or some such, possibly with sudo).

## Building the Documents

You can build the thesis as a document with the following commands

```bash
latexmk -pdf
```