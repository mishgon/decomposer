"""Docker-free Decomposer integration for the internal GAIA2/ARE runner.

The package stays import-light so the Decomposer service never imports ARE.
The Gaia benchmark runtime loads ``gyms.gaia2.plugin:create_plugin`` explicitly.
"""
