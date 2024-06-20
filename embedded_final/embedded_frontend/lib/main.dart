import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image1;
  File? _image2;
  var _similarityResult;
  final ImagePicker _picker = ImagePicker();

  Future<void> _getImageFromGallery(ImageSource source, int imageNumber) async {
    final pickedFile = await _picker.pickImage(source: source);

    setState(() {
      if (pickedFile != null) {
        if (imageNumber == 1) {
          _image1 = File(pickedFile.path);
        } else {
          _image2 = File(pickedFile.path);
        }
      } else {
        print('No image selected.');
      }
    });
  }

  Future<void> _uploadImages() async {
    if (_image1 == null || _image2 == null) {
      print('Please select both Image 1 and Image 2.');
      return;
    }

    String base64Image1 = base64Encode(_image1!.readAsBytesSync());
    String base64Image2 = base64Encode(_image2!.readAsBytesSync());

    final uri = Uri.parse('http://10.0.2.2:5000/process_images');
    try {
      final response = await http.post(
        uri,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"image1": base64Image1, "image2": base64Image2}),
      );

      if (response.statusCode == 200) {
        final decodedResponse = jsonDecode(response.body);
        setState(() {
          _similarityResult = decodedResponse;
        });

        _showResultDialog(decodedResponse);
      } else {
        print('Failed to upload images: ${response.statusCode}');
      }
    } catch (e) {
      print('Error uploading images: $e');
    }
  }

  void _showResultDialog(Map<String, dynamic> similarityResponse) {
    List<Widget> children = [];

    // Euclidean Distance
    double euclideanDistance = similarityResponse["Euclidean Distance"];
    children.add(Text('Euclidean Distance: \n$euclideanDistance'));

    // Cosine Similarity
    double cosineSimilarity = similarityResponse["Cosine Similarity"];
    children.add(Text('Cosine Similarity: \n$cosineSimilarity'));

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Similarity Result'),
          content: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: children,
          ),
          actions: <Widget>[
            TextButton(
              child: Text('OK'),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  void _resetImages() {
    setState(() {
      _image1 = null;
      _image2 = null;
      _similarityResult = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Image Similarity Calculation'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image1 == null
                ? Text('No image selected for Image 1.')
                : Image.file(_image1!),
            SizedBox(height: 20),
            _image2 == null
                ? Text('No image selected for Image 2.')
                : Image.file(_image2!),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _uploadImages,
              child: Text('Upload Images and Calculate Similarity'),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _resetImages,
              child: Text('Reset Images'),
            ),
          ],
        ),
      ),
      floatingActionButton: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: <Widget>[
          FloatingActionButton(
            onPressed: () => _getImageFromGallery(ImageSource.gallery, 1),
            tooltip: 'Pick Image 1',
            child: Icon(Icons.add_a_photo),
          ),
          SizedBox(height: 16),
          FloatingActionButton(
            onPressed: () => _getImageFromGallery(ImageSource.gallery, 2),
            tooltip: 'Pick Image 2',
            child: Icon(Icons.add_a_photo),
          ),
        ],
      ),
    );
  }
}
