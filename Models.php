<?php
$models = file_exists('models')?scandir('models'):[];
header('Content-Type: application/json');
if(empty($models)){
  echo '{"success":false,"error":"No hay modelos disponibles"}';
}else{
  echo json_encode($models);
}
?>