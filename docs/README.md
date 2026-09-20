# Documentation site

Built with [DocFX](https://dotnet.github.io/docfx/) (MIT): Markdown pages here plus the API
reference generated from the libraries' XML doc comments.

```powershell
dotnet tool install -g docfx        # once
docfx docs/docfx.json --serve       # builds docs/_site and serves it at http://localhost:8080
```

`docs/_site` and `docs/api` are generated and git-ignored. The GitHub Actions workflow
`.github/workflows/docs.yml` builds and publishes the site to GitHub Pages on every push to `main`.
