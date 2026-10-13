"""Docker-free Decomposer integration for the internal GAIA2/ARE runner.

The package stays import-light so the Agent Server never imports ARE.
The Gaia benchmark runtime loads ``gyms.gaia2.are:create_plugin`` explicitly.
"""
