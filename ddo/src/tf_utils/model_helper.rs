/// example based on https://stackoverflow.com/a/68567498 and https://medium.com/@alfred.weirich/introduction-to-working-with-tensorflow-for-rust-e8ed82853e62

use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};
use crate::Decision;
use std::path::Path;

pub struct TfModel{
    graph: Graph,
    bundle: SavedModelBundle,
    input_name: String,
    output_name: String,
}

impl TfModel {
    pub fn new<P: AsRef<Path>>(model_path:P,input_name:String,output_name:String) -> Self {
        // let mut graph = Graph::new();
        // let mut bundle = SavedModelBundle::new();
        let (graph,bundle) = Self::load_model(model_path);

        TfModel{
            graph, 
            bundle, 
            input_name,
            output_name}
    }

    //TODO who creates the initial model
    pub fn load_model<P: AsRef<Path>>(model_path: P) -> (Graph,SavedModelBundle) {
        // Initialize an empty graph
        let mut graph = Graph::new();
        // Load saved model bundle (session state + meta_graph data)
        let bundle = 
            SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, model_path)
            .expect("Can't load saved model");
        (graph,bundle)
    }
}
pub trait ModelHelper{

    type State;
    // type OutputTensor where Self::OutputTensor: Copy;
    // type OutputTensor;

    fn state_to_input_tensor(&self,state: &Self::State) -> Result<Tensor<f32>, tensorflow::Status>;

    fn perform_inference(&self, model:&TfModel, state: &Self::State) -> Tensor<f32> {
        let signature_input_parameter_name = &model.input_name;
        let signature_output_parameter_name = &model.output_name;

        // initialise inout tensor
        let tensor: Tensor<f32> = self.state_to_input_tensor(state).unwrap();

        // Get the session from the loaded model bundle
        let session = &model.bundle.session;

        // Get signature metadata from the model bundle
        //TO DO: Unclear what reusing a session over and over does here tbh -  should we somehow "clear" and restart it?
        // Do any states in the model change due to inference? - PS: It shouldnt!
        let signature = model.bundle
            .meta_graph_def()
            .get_signature("serving_default")
            .unwrap();

        // Get input/output info
        let input_info = signature.get_input(&signature_input_parameter_name).unwrap();
        let output_info = signature.get_output(&signature_output_parameter_name).unwrap();

        // Get input/output ops from graph
        let input_op = model.graph
            .operation_by_name_required(&input_info.name().name)
            .unwrap();
        let output_op = model.graph
            .operation_by_name_required(&output_info.name().name)
            .unwrap();
        
        // Manages inputs and outputs for the execution of the graph
        let mut args = SessionRunArgs::new();
        args.add_feed(&input_op, 0, &tensor); // Add any inputs

        let out = args.request_fetch(&output_op, 0); // Request outputs

        // Run model
        session.run(&mut args) // Pass to session to run
            .expect("Error occurred during calculations");

        // Fetch outputs after graph execution
        let out_res = args.fetch(out).unwrap();

        return out_res;
    }

    fn extract_decision_from_model_output(&self, state: &Self::State, output: Tensor<f32>) -> Option<Decision>;
}