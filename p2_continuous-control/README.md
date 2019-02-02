These control values are applied as torques to the bodies making up the arm:

```cs
public override void AgentAction(float[] act)
{
    float torque_x = Mathf.Clamp(act[0], -1, 1) * 100f;
    float torque_z = Mathf.Clamp(act[1], -1, 1) * 100f;
    rbA.AddTorque(new Vector3(torque_x, 0f, torque_z));

    torque_x = Mathf.Clamp(act[2], -1, 1) * 100f;
    torque_z = Mathf.Clamp(act[3], -1, 1) * 100f;
    rbB.AddTorque(new Vector3(torque_x, 0f, torque_z));
}
```

By default the output from our provided PPO algorithm pre-clamps the values of vectorAction into the [-1, 1] range.
It is a best practice to manually clip these as well, if you plan to use a 3rd party algorithm with your environment.
As shown above, you can scale the control values as needed after clamping them.