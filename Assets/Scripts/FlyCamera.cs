using UnityEngine;
using System.Collections;
 
public class FlyCamera : MonoBehaviour {
 
    /*
        wasdqe : basic movement
        shift : Makes camera accelerate
        space : Moves camera on X and Z axis only.  So camera doesn't gain any height
    */

    public bool lookAtOrigin = false;
    public float mainSpeed = 0.5f; //regular speed

    private float shiftMultiplier = 2.0f; //multiplied by how long shift is held.  Basically running
    private float maxShift = 1000.0f; //Maximum speed when holdin gshift
    private float camSens = 0.25f; //How sensitive it with mouse
    private Vector3 lastMouse = new Vector3(255, 255, 255); //kind of in the middle of the screen, rather than at the top (play)
     
    void Update () {
       
        //Keyboard commands
        Vector3 p = GetBaseInput();
        if (p.sqrMagnitude > 0){ // only move while a direction key is pressed
          if (Input.GetKey (KeyCode.LeftShift)){
              p  = p * shiftMultiplier;
              p.x = Mathf.Clamp(p.x, -maxShift, maxShift);
              p.y = Mathf.Clamp(p.y, -maxShift, maxShift);
              p.z = Mathf.Clamp(p.z, -maxShift, maxShift);
          } else {
              p = p * mainSpeed;
          }
         
          p = p * Time.deltaTime;
          Vector3 newPosition = transform.position;
          if (Input.GetKey(KeyCode.Space)){ //If player wants to move on X and Z axis only
              transform.Translate(p);
              newPosition.x = transform.position.x;
              newPosition.z = transform.position.z;
              transform.position = newPosition;
          } else {
              transform.Translate(p);
          }
        }

        if (lookAtOrigin)
        {
            transform.LookAt(new Vector3(0, 0, 0));
        }
        else
        {
            lastMouse = Input.mousePosition - lastMouse;
            lastMouse = new Vector3(-lastMouse.y * camSens, lastMouse.x * camSens, 0);
            lastMouse = new Vector3(transform.eulerAngles.x + lastMouse.x, transform.eulerAngles.y + lastMouse.y, 0);
            transform.eulerAngles = lastMouse;
            lastMouse = Input.mousePosition;
        }
    }
     
    private Vector3 GetBaseInput() { //returns the basic values, if it's 0 than it's not active.
        Vector3 p_Velocity = new Vector3();
        if (Input.GetKey (KeyCode.W)){
            p_Velocity += new Vector3(0, 0 , 1);
        }
        if (Input.GetKey (KeyCode.S)){
            p_Velocity += new Vector3(0, 0, -1);
        }
        if (Input.GetKey (KeyCode.A)){
            p_Velocity += new Vector3(-1, 0, 0);
        }
        if (Input.GetKey (KeyCode.D)){
            p_Velocity += new Vector3(1, 0, 0);
        }
        if (Input.GetKey (KeyCode.Q)){
            p_Velocity += new Vector3(0, 1, 0);
        }
        if (Input.GetKey (KeyCode.E)){
            p_Velocity += new Vector3(0, -1, 0);
        }
        return p_Velocity;
    }
}