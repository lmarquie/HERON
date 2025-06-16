────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.10/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:121 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.10/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:645 in code_to_exec                                     

                                                                                

  /mount/src/heron/app.py:62 in <module>                                        

                                                                                

      59 clear_caches()                                                         

      60                                                                        

      61 # Theme configuration                                                  

  ❱   62 st.set_page_config(                                                    

      63 │   page_title="Herbert Insight AI",                                   

      64 │   layout="wide",                                                     

      65 │   initial_sidebar_state="expanded",                                  

                                                                                

  /home/adminuser/venv/lib/python3.10/site-packages/streamlit/runtime/metrics_  

  util.py:444 in wrapped_func                                                   

                                                                                

  /home/adminuser/venv/lib/python3.10/site-packages/streamlit/commands/page_co  

  nfig.py:273 in set_page_config                                                

                                                                                

  /home/adminuser/venv/lib/python3.10/site-packages/streamlit/runtime/scriptru  

  nner_utils/script_run_context.py:192 in enqueue                               

────────────────────────────────────────────────────────────────────────────────

StreamlitSetPageConfigMustBeFirstCommandError: `set_page_config()` can only be 

called once per app page, and must be called as the first Streamlit command in 

your script.


For more information refer to the 

[docs](https://docs.streamlit.io/develop/api-reference/configuration/st.set_page

_config).

2025-06-16 15:21:28.674 `label` got an empty value. This is discouraged for accessibility reasons and may be disallowed in the future by raising an exception. Please provide a non-empty label and hide it with label_visibility if needed.

main
lmarquie/heron/main/app.py
