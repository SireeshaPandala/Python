package rashmi.umkc.edu.insertcalendarevent;

import java.util.Calendar;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.provider.CalendarContract;
import android.support.annotation.NonNull;
import android.view.Menu;
import android.view.View;
import android.widget.Button;
import android.widget.CalendarView;
import android.widget.TextView;

import com.example.insertcalendareventintent.R;

public class MainActivity extends Activity {

    CalendarView calendarView;
    TextView myDate;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button butinsert = (Button) findViewById(R.id.create_eventbut);
        butinsert.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                // TODO Auto-generated method stub
                insert();
            }
        });
        calendarView = (CalendarView)findViewById(R.id.mycalender);
        myDate = (TextView)findViewById(R.id.displayDate);

        calendarView.setOnDateChangeListener(new CalendarView.OnDateChangeListener() {
            @Override
            public void onSelectedDayChange(@NonNull CalendarView calendarView, int i, int i1, int i2) {
                String date = (i1+1)+ "-" + i2 + "-" + i;
                myDate.setText(date);
            }
        });
    }


    @SuppressLint("NewApi")
    public void insert() {
        Intent intent = new Intent(Intent.ACTION_INSERT,
                CalendarContract.Events.CONTENT_URI);
        // Add the calendar event details
        intent.putExtra(CalendarContract.Events.TITLE, "ISL Exam");
        intent.putExtra(CalendarContract.Events.DESCRIPTION,
                "Project Submission and Final Exam");
        intent.putExtra(CalendarContract.Events.EVENT_LOCATION,
                "Cockefair Hall");
        Calendar startTime = Calendar.getInstance();
        startTime.set(2018, 12, 11);
        intent.putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME,
                startTime.getTimeInMillis());
        intent.putExtra(CalendarContract.EXTRA_EVENT_ALL_DAY, true);
        // Use the Calendar app to add the new event.
        startActivity(intent);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }

}