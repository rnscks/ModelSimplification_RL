from src.model_3d.cad_model import Assembly, AssemblyFactory,ViewDocument


assembly: Assembly = AssemblyFactory.create_assembly("AirCompressor.stp")
view_document = ViewDocument()  
assembly.add_to_view_document(view_document)
view_document.display()