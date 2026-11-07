# Deployment examples

These files are reference templates, not production configuration:

- `nginx.conf.example`: static page routes and response headers.
- `origin-guard.conf.example`: request-origin checks and fixed login return paths.
- `compose.embed.yaml.example`: example settings for an embedded chat service.

Replace example domains, filesystem paths and role names for your environment.
Review the templates alongside your existing authentication, HTTPS and proxy
configuration before use. Keep actual credentials, environment files and
production operations notes outside this public repository.
