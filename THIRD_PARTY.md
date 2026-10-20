# Third party code

`src/lavis/` contains the server's vendored and locally modified copy of [Salesforce LAVIS](https://github.com/salesforce/LAVIS). Its upstream BSD 3-Clause license is preserved in `licenses/LAVIS-LICENSE.txt`. Individual files retain their copyright and license headers, including references to other upstream implementations.

The CLIP tokenizer implementation identifies its origin as [OpenAI CLIP](https://github.com/openai/CLIP), originally under the MIT license. The bundled tokenizer vocabulary is retained because LAVIS imports it during initialization. The original CLIP license is included in `licenses/CLIP-LICENSE.txt`.

The Conceptual Captions downloader retains its bundled `LICENSE` in its original directory. This code release does not grant additional redistribution rights for model weights or datasets; those files are maintained separately from the source repository.
