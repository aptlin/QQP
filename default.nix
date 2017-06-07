with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "env";

  # Mandatory boilerplate for buildable env
  env = buildEnv { name = name; paths = buildInputs; };
  builder = builtins.toFile "builder.sh" ''
    source $stdenv/setup; ln -s $env $out
  '';

  # Customizable development requirements
buildInputs = [
    # Add packages from nix-env -qaP | grep -i needle queries
    autojump
    emacs
    git
    
    # With Python configuration requiring a special wrapper
    (python35.buildEnv.override {
      ignoreCollisions = true;
      extraLibs = with python35Packages; [
        # Add pythonPackages without the prefix
	h5py
	pytest
	nltk
	pyenchant
	scipy
	ipython        
	ipykernel
        numpy
	nltk
    #jupyter
	jupyter_client
    notebook
	#qtconsole
	jupyter_console
        toolz
	seaborn
        pandas
        numpy
        scikitlearn
        tqdm
	more-itertools        
        Keras
	#spacy
	gensim
        tensorflow
	tflearn
        python-Levenshtein
        setuptools
	matplotlib
	xgboost
	networkx
	flake8
	mlxtend
	];
    })
  ];

  # Customizable development shell setup with at last SSL certs set
  shellHook = ''
  source ~/.bash_profile
  '';
}
